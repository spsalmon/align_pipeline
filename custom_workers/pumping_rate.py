import argparse
import os
import pickle

import numpy as np
import polars as pl
import scipy
import skimage.filters
from align_toolbox.foundation.file_handling import extract_time_point, write_filemap
from align_toolbox.foundation.image_handling import normalize_image, read_tiff_file
from joblib import Parallel, delayed
from scipy.signal import find_peaks
from whittaker_eilers import WhittakerSmoother


def load_pickles(*pickle_paths):
    loaded_pickles = []
    for pickle_path in pickle_paths:
        with open(pickle_path, "rb") as f:
            files = pickle.load(f)
        loaded_pickles.append(files)
    return loaded_pickles


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filemap", type=str, help="pickled filemap")
    parser.add_argument("-b", "--block_config", type=str, help="pickled block config")
    parser.add_argument("-c", "--config", type=str, help="pickled config")
    parser.add_argument("-o", "--output", type=str, help="output")
    parser.add_argument(
        "--input_column", type=str, help="column in filemap to use as input"
    )
    parser.add_argument(
        "--frame_rate", type=float, help="frame rate of the input movies"
    )
    parser.add_argument("--height", type=int, help="minimum height for peak detection")
    parser.add_argument("--width", type=int, help="minimum width for peak detection")
    parser.add_argument(
        "--std_coeff",
        type=float,
        help="coefficient of standard deviation for peak detection prominence",
    )
    parser.add_argument(
        "--dist", type=int, help="minimum distance between peaks in frames"
    )
    return parser.parse_args()


def process_movie(
    movie_path,
    std_coeff,
    distance,
    width,
    height,
    frame_rate,
    time_regex=r"Time(\d+)",
    point_regex=r"Point(\d+)",
):

    try:
        time, point = extract_time_point(movie_path, time_regex, point_regex)
    except ValueError:
        return None

    movie = read_tiff_file(movie_path)
    pumping_rate = get_pumping_rate(
        movie, std_coeff, distance, width, height, frame_rate
    )

    row = {"Time": time, "Point": point, "pumping_rate": pumping_rate}
    return row


def get_pumping_rate(movie, std_coeff, distance, width, height, frame_rate):
    if np.sum(movie) == 0:
        return None

    movie = normalize_image(movie)
    std_fluo = np.std(movie, axis=(1))
    sum_fluo = np.sum(movie, axis=(1))
    sum_fluo = sum_fluo - np.mean(sum_fluo, axis=1, keepdims=True)

    # gaussian filter for each frame
    sum_fluo = [skimage.filters.gaussian(frame, sigma=3) for frame in sum_fluo]
    # sum_fluo = skimage.filters.gaussian(sum_fluo, sigma=3)
    sum_fluo = np.array(sum_fluo)

    middle = sum_fluo.shape[1] // 2
    bulb_positions = []
    pumping_measures = []
    for i in range(sum_fluo.shape[0]):
        peaks, _ = find_peaks(sum_fluo[i], distance=10, height=np.mean(sum_fluo[i]))

        # look for the furthest peaks from the middle
        distances = np.abs(peaks - middle)
        sorted_indices = np.argsort(distances)
        sorted_peaks = peaks[sorted_indices]
        if len(sorted_peaks) < 2:
            return None
        bulb = sorted_peaks[-1]
        pumping_metric = -std_fluo[i][bulb]
        bulb_positions.append(bulb)
        pumping_measures.append(pumping_metric)

    # detrend the pumping measures
    trend = scipy.signal.savgol_filter(pumping_measures, window_length=101, polyorder=5)
    pumping_measures = pumping_measures - trend

    whittaker_smoother = WhittakerSmoother(
        lmbda=0.25, order=2, data_length=len(pumping_measures)
    )

    pumping_measures = whittaker_smoother.smooth(pumping_measures)

    pumping_measures = np.array(pumping_measures) + np.abs(np.min(pumping_measures))

    peaks, _ = find_peaks(
        pumping_measures,
        distance=distance,
        prominence=std_coeff * np.std(pumping_measures),
        width=width,
        height=height,
    )

    return len(peaks) / (len(pumping_measures) / frame_rate)


def main():
    args = get_args()
    filemap = load_pickles(args.filemap)[0]
    config = load_pickles(args.config)[0]

    time_regex = config.get("time_regex", r"Time(\d+)")
    point_regex = config.get("point_regex", r"Point(\d+)")

    output = args.output
    input_column = args.input_column
    frame_rate = args.frame_rate
    height = args.height
    width = args.width
    dist = args.dist
    std_coeff = args.std_coeff

    output_name = os.path.basename(output).split(".")[0]

    input_files = pl.Series(filemap.select(pl.col(input_column))).to_list()

    print(f"Processing {len(input_files)} movies for pumping rate estimation.")

    rows = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(process_movie)(
            movie_path,
            std_coeff,
            dist,
            width,
            height,
            frame_rate,
            time_regex,
            point_regex,
        )
        for movie_path in input_files
    )

    df = pl.DataFrame(rows)
    # rename pumping_rate column to include output name
    df = df.rename({"pumping_rate": output_name})

    # sort by Time and Point
    df = df.sort(["Time", "Point"])

    write_filemap(df, output)


if __name__ == "__main__":
    main()
