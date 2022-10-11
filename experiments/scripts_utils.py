def load_true_file(filepath):
    with open(filepath) as file:
        Y = []
        for i, line in enumerate(file):
            if i == 0 and len(line.split(' ')) == 3:
                continue
            Y.append([int(y) for y in line.strip().split(' ', 1)[0].split(',') if ':' not in y])
        return Y


def load_pred_file(filepath):
    with open(filepath) as file:
        Y = []

        def convert_y(y):
            y = y.split(':')
            if len(y) == 2:
                return (int(y[0]), float(y[1]))
            else:
                return int(y)

        for line in file:
            Y.append([convert_y(y) for y in line.strip().split(' ')])
        return Y


def load_weights_file(filepath):
    with open(filepath) as file:
        v = []
        for line in file:
            v.append(float(line.strip()))
        return v
