import os
import aiohttp
import asyncio
import xarray as xr
import aiofiles
import time
import numpy as np


base_url = "https://portal.nccs.nasa.gov/datashare/gmao/geos-cf/v1/das/"
output_dir = "/project/dilkina_565/aerosol_data_full/"



def days_per_month_calc(y=int, m=int):
    days_per_month = [31,28,31,30,31,30,31,31,30,31,30,31]
    days_per_month_leap = [31,29,31,30,31,30,31,31,30,31,30,31]
    if y%4 == 0:
        return days_per_month_leap[m-1]
    else:
        return days_per_month[m-1]

async def download_and_extract_variable(session, url, file_path, y, m, d, h, min, sem, max_retries=5):
    retries = 0
    while retries < max_retries:
        async with sem:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        temp_file = file_path + ".temp.nc4"
                        async with aiofiles.open(temp_file, 'wb') as file:
                            await file.write(content)

                        with xr.open_dataset(temp_file) as ds:
                            extract = ds['PM25_RH35_GCC'] #for nc4
                            extracted_data = ds['PM25_RH35_GCC'].values #for numpy
                            xarray_file_path = file_path
                            extract.to_netcdf(xarray_file_path)
                            print(f"xarray data saved to {xarray_file_path}")

                        numpy_dir = os.path.join(output_dir, 'numpy', f"{y}", f"M{m:02d}")
                        os.makedirs(numpy_dir, exist_ok=True)
                        
                        npy_file_name = f"D{d:02d}_{h:02d}{min:02d}.npy"
                        npy_file_path = os.path.join(numpy_dir, npy_file_name)
                        np.save(npy_file_path, extracted_data)
                        print(f"Data saved as NumPy to {npy_file_path}")

                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                        return npy_file_path
                    else:
                        print(f"Failed to download the file{file_path}. Status code: {response.status}")
                        retries += 4
            except Exception as e:
                print(f"An error occurred: {e}. For URL: {url}")

            retries += 1
            if retries < max_retries:
                print(f"Retrying... Attempt {retries + 1} of {max_retries}")
            if retries == max_retries:
                print(f"Failed to download after {max_retries} attempts. For URL: {url}")

        return None

async def file_traversal():
    sem = asyncio.Semaphore(5)
    tasks = []
    async with aiohttp.ClientSession() as session:
        for y in range(2018, 2023+1):
            for m in range(1, 12+1):
                for d in range(1, days_per_month_calc(y, m) + 1):
                    for h in range(0, 24):
                        min = 30
                        curr_dir = os.path.join(output_dir, str(y), f"M{m:02d}")
                        os.makedirs(curr_dir, exist_ok=True)
                        url = f"{base_url}Y{y}/M{m:02d}/D{d:02d}/GEOS-CF.v01.rpl.aqc_tavg_1hr_g1440x721_v1.{y}{m:02d}{d:02d}_{h:02d}{min:02d}z.nc4"
                        file_path = os.path.join(curr_dir, f"D{d:02d}_{h:02d}{min:02d}.nc4")
                        if not os.path.exists(file_path):
                            task = asyncio.create_task(download_and_extract_variable(session, url, file_path, y, m, d, h, min, sem))
                            tasks.append(task)
                        else:
                            print(f"File already exists: {file_path}")
        results = await asyncio.gather(*tasks)
        return results

if __name__ == '__main__':
    start = time.time()
    asyncio.get_event_loop().set_debug(True)
    asyncio.run(file_traversal())
    end = time.time()
    print(f"Time elapsed: {end - start} seconds")