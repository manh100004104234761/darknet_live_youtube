
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
                help="Path to the plate image")
ap.add_argument("-f", "--folder", default=None,
                help="Path to the folder plate images")
args = vars(ap.parse_args())

if __name__ == '__main__':
    print(args['image'])
    print(args['folder'])
