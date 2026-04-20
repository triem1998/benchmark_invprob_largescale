import sys
import argparse
import os
import numpy as np
import urllib.request
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp


def createParser():
    '''Create command line interface.'''
    # When --help or no args are given, print this help
    usage_text = (
        "Run python with this script:"
        "  python " + __file__ + "[options]"
    )

    parser = argparse.ArgumentParser(description=usage_text)

    parser.add_argument(
        "--data_release", default="dr3",
        choices=["dr2", "dr3"],
        help="Data release",
    )
    parser.add_argument(
        "--ra", default=180.0,
        type=float,
        help="Right Ascension in degrees",
    )
    parser.add_argument(
        "--dec", default=45.0,
        type=float,
        help="Declination in degrees",
    )
    parser.add_argument(
        "--fits_number", default=1,
        type=int,
        help="Number of FITS files to download",
    )
    parser.add_argument(
        "--fits_size", default=4096,
        type=int,
        help="Image size in the FITS file",
    )
    parser.add_argument(
        "--random", default=0,
        choices=[0, 1],
        type=int,
        help="Whether to download images with random coordinates (1) or with the specified RA and DEC (0)",
    )
    parser.add_argument(
        "--output_path", default="dataset",
        type=str,
        help="Output path for downloaded FITS files",
    )

    return parser


def download(data_release, ra, dec, fits_size, output_path):

    fov = image_to_fov(fits_size)

    url = f'https://lofar-surveys.org/{data_release}-cutout.fits?pos={ra}%20{dec}&size={fov}'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ra = round(ra, 4)
    dec = round(dec, 4)

    file_Path = os.path.join(output_path, 'RA_{ra}_DEC_{dec}_size_{fits_size}.fits'.format(ra=ra, dec=dec, fits_size=fits_size))
    print(f"Downloading FITS file RA={ra}, DEC={dec} to {file_Path}...")
    try:
        urllib.request.urlretrieve(url, file_Path)
        print(f"Successfully downloaded {file_Path}")
        return 0, file_Path
    except Exception as e:
        print(f"FITS file RA={ra}, DEC={dec} doesn't exist...")
        return 1, None

def image_to_fov(image_size):
    fov = float(image_size / 40)
    return fov

def generate_random_coordinates():

    ra = np.random.uniform(0, 360)
    dec = np.random.uniform(-90, 90)
    return ra, dec

def get_mosaic_coords(fov):
    mosaic_coords = []
    fov_deg = fov / 60
    ra1, dec1 = generate_random_coordinates()
    ra2, dec2 = (fov_deg / np.cos(np.deg2rad(dec1))), dec1
    ra3, dec3 = ra1, dec1-fov_deg
    ra4, dec4 = (fov_deg / np.cos(np.deg2rad(dec3))), dec3

    mosaic_coords.extend([(ra1, dec1), (ra2, dec2), (ra3, dec3), (ra4, dec4)])
    return mosaic_coords

def download_mosaic(data_release, fits_size, output_path):
    
    paths = []
    mosaic_fits_size = int(fits_size / 4) 
    mosaic_fov = image_to_fov(mosaic_fits_size)
    mosaic_coords = get_mosaic_coords(mosaic_fov)

    for ra, dec in mosaic_coords:
        status, file_Path = download(data_release, ra, dec, mosaic_fits_size, output_path)
        if status == 0:
            paths.append(file_Path)
        else:
            print(f"Failed to download FITS file for RA={ra}, DEC={dec}. Skipping this tile.")
    return status, paths

def make_mosaic(mosaic_paths, output_path):

    data_list = []
    wcs_list = []

    # Lire toutes les images
    for path in mosaic_paths:
        with fits.open(path) as hdul:
            data = hdul[0].data.astype(float)
            wcs = WCS(hdul[0].header)

            data_list.append(data)
            wcs_list.append(wcs)

    target_wcs = wcs_list[0]
    shape_out = data_list[0].shape

    # accumulation + poids
    mosaic = np.zeros(shape_out)

    for data, wcs in zip(data_list, wcs_list):

        reprojected, _ = reproject_interp(
            (data, wcs),
            target_wcs,
            shape_out=shape_out
        )

        mask = np.isfinite(reprojected)

        mosaic[mask] += reprojected[mask]

    header = target_wcs.to_header()

    out_path = os.path.join(output_path, "mosaic.fits")
    fits.PrimaryHDU(data=mosaic, header=header).writeto(out_path, overwrite=True)

    print(f"Mosaic created: {out_path}")
    return out_path


def run(args):

    if args.random == 1:
        fits_number = args.fits_number
        k = 0
        pos = []
        while k < fits_number:
            ra, dec = generate_random_coordinates()
            if round(ra, 4) in [round(p[0], 4) for p in pos] and round(dec, 4) in [round(p[1], 4) for p in pos]:
                print("Coordinates already used, generating new random coordinates...")
                fits_number += 1
                continue
            pos.append((ra, dec))
            status, _ = download(args.data_release, ra, dec, args.fits_size, output_path=args.output_path)

            if status != 0:
                print("Downloading another FITS file with random coordinates...")
                print("...............................................................................")
                fits_number += 1
            k += 1
    else:
        ra, dec = args.ra, args.dec
        if ra < 0 or ra >= 360:
            raise ValueError("RA must be in the range [0, 360)")
        if dec < -90 or dec > 90:
            raise ValueError("DEC must be in the range [-90, 90]")
        status = download(args.data_release, ra, dec, args.fits_size, output_path=args.output_path)
        if status != 0:
            raise ValueError(f"FITS file RA={ra}, DEC={dec} doesn't exist.")
        
def run_mosaic(args):

    fits_number = args.fits_number
    k = 0
    while k < fits_number:
        status, mosaic_paths = download_mosaic(args.data_release, args.fits_size, args.output_path)
        if status != 0:
            print("...............................................................................")
            fits_number += 1
            continue
        make_mosaic(mosaic_paths, output_path=args.output_path)
   

def main():
    argv = sys.argv

    parser = createParser()
    args = parser.parse_args(argv[1:])

    run(args)

if __name__ == "__main__":

    err = 1
    try:
        err = main()
    except Exception as e:
        print("\n" + str(e))
        sys.exit(err)
    sys.exit(err)
