import sys
import argparse
from subprocess import call

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fnumbers", type=str, nargs="*"              )
    parser.add_argument("-r"       , type=int                         )
    parser.add_argument("-t"   , type=str,            default="raw")
    args     = parser.parse_args(sys.argv[1:])

    print(f'filenames = {args.fnumbers}, run = {args.r}, type = {args.t}')

    adrr = "icuser@195.77.159.50:"
    tag  = "v0.9.6-29-g2ec1284_20180607_kr"
    kdst = "dev"

    if args.t == "raw":
        for number in args.fnumbers:
            filename =f"run_{args.r}_{number}_waveforms.h5"
            dest = f"./data/{args.r}/{filename}"
            print(f'copying file = {filename} to {dest}')
            path= f"{adrr}/analysis/{args.r}/hdf5/{args.d}/{filename}"
            call(["scp", "-P 6030", path, dest])

    elif args.t == "kdst":
        filename =f"kdst_{args.r}_{tag}_{kdst}.h5"
        dest = f"./kdst/{args.r}/kdst_{args.r}.h5"
        print(f'copying file = {filename} to {dest}')
        path= f"{adrr}/analysis/{args.r}/hdf5/{filename}"
        call(["scp", "-P 6030", path, dest])
    else:
        print("argument type must be 'raw', or 'kdst' ")
        sys.exit()
