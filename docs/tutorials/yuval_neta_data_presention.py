import glob
import pickle
from matplotlib import pyplot as plt
import numpy

def isRotationMatrix(R):
    # square matrix test
    if R.ndim != 3 or R.shape[0] != R.shape[1]:
        return False
    should_be_identity = numpy.allclose(R.dot(R.T), numpy.identity(R.shape[0], numpy.float64))
    should_be_one = numpy.allclose(numpy.linalg.det(R), 1)
    return should_be_identity and should_be_one

data_train_paths = [f for f in glob.glob(r'/home/neta-katz@staff.technion.ac.il/Downloads/*.pkl')]
for path in data_train_paths:
    with open(path, 'rb') as outfile:
        x = pickle.load(outfile)
        print(str(x.keys()))

        # show images
        for img in x["images"]:
            print(img.max())
            print(img.min())
            print("------------------------------")
            plt.imshow(img, interpolation='nearest', cmap='gray')
            plt.show()
        # show cameras positions
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xdata = list()
        ydata = list()
        zdata = list()
        vectors = list()
        for pos, r_mat in zip(x["cameras_pos"], x["cameras_R"]):
            if not isRotationMatrix(r_mat):
                print("bad rotation matrix")
            xdata.append((pos[0]))
            ydata.append((pos[1]))
            zdata.append((pos[2]))
            vec = r_mat.dot(numpy.array([0, 0, 500]))
            vectors.append([pos[0], pos[1], pos[2], vec[0], vec[1], vec[2]])
            print(x["cameras_R"][0])
        ax.scatter(xdata, ydata, zdata, c='r', marker='o')
        soa = numpy.array(vectors)
        # soa[3:] *= 0.5
        # X, Y, Z, U, V, W = soa[:]
        # ax.quiver(soa[:,0], soa[:,1], soa[:,2], soa[:,3], soa[:,4] ,soa[:,5], normalize=False)
        ax.scatter(soa[:,0] + soa[:,3], soa[:,1] + soa[:,4], soa[:,2] + + soa[:,5], c='g', marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

        print("not sure of : cameras_k, cameras_P, grid(?), ext, mask, mask_morph")