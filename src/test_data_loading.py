import xarray as xr
import glob
import os

DATA_DIR = "data"
TMAX_FILES_PATTERN = os.path.join(DATA_DIR, "gfdl-esm4_r1i1p1f1_w5e5_historical_tasmax_global_daily_*.nc")

def test_load_individual_files():
    print("--- Testing individual TMAX file opening ---")
    tmax_files = sorted(glob.glob(TMAX_FILES_PATTERN))
    if not tmax_files:
        print("No TMAX files found!")
        return

    all_files_ok = True
    for f in tmax_files:
        print(f"--- Testing file: {f} ---")
        try:
            # Open the dataset without loading data into memory
            with xr.open_dataset(f, chunks={}) as ds:
                print(f"Successfully opened: {f}")
                print("Dataset details:")
                print(ds)
                print("-------------------------")
        except Exception as e:
            print(f"!!! FAILED to open or read metadata from: {f}")
            print(f"Error: {e}")
            all_files_ok = False
            print("-------------------------")

    if all_files_ok:
        print("\n--- All files were opened successfully (metadata only). ---")
    else:
        print("\n--- One or more files failed to open. ---")

if __name__ == "__main__":
    test_load_individual_files()