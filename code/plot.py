import json
import argparse
import matplotlib.pyplot as plt

def read_and_plot(json_file):
    data = json.load(json_file)
    x = list(range(len(data['durations'])))
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 8))


    ax1.plot(x, data['durations'])
    ax1.set_title('durations')

    ax2.plot(x, data['losses'])
    ax2.set_title('losses')

    ax3.plot(x, data['rewards'])
    ax3.set_title('rewards')

    plt.show()


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    parser.add_argument('--file', type=str, required=True, help='file path of run data')

    config = parser.parse_args()

    with open(config.file) as json_file:
      read_and_plot(json_file)