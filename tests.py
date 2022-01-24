import matplotlib.pyplot as plt

from utils import OrnsteinUhlenbeck

def OrnsteinUhlenbeck_visualization():
    gen = OrnsteinUhlenbeck(theta=0.1, sigma=0.1)
    l = []
    for _ in range(100):
        l.append(gen.step().item())

    plt.plot(l)
    plt.show()


if __name__ == "__main__":
    OrnsteinUhlenbeck_visualization()