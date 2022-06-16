import cv2
import os
import inspect
import sys


class Prep:
    """
    Class read file from path using preprocessed methods by attributes
    """

    def __init__(self, image):
        try:
            self.image = cv2.imread(image) if os.path.isfile(image) else 'Non img file'
            self.origin = self.image.copy()
            self.gray = cv2.imread(image, 0)
        except:
            print("Image file not found, check path!")
            sys.exit()

    def get_info(self, image, show=False):
        var = self.retrieve_name(image).upper()
        print(f"VAR:\t{var}"
              f"\nDTYPE:\t{image.dtype}"
              f"\nTYPE:\t{type(image)}"
              f"\nN_DIM:\t{image.ndim}"
              f"\nSHAPE:\t{image.shape}\n")
        if show:
            cv2.imshow(var, image)
            cv2.waitKey(2000)
            cv2.destroyWindow(var)

    def preprocess(self, prep=None, clr=None, ch=None, dt=None):
        image = self.image
        if prep == 'resize':
            image = cv2.resize(self.image, None, fx=0.8, fy=0.8)
            # image = cv2.resize(image, (200, 200))
        elif prep == 'flip':
            image = cv2.flip(image, 0)
        elif prep == 'uflip':
            image = cv2.flip(image, 1)

        if clr == 'gray':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif clr == 'rgb':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif clr == 'hsv':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if ch == 1 and image.ndim > 1:
            image = image[:, 0, 0]
        elif ch == 2 and image.ndim > 2:
            image = image[:, :, 0]
        elif ch == 3 and image.ndim <= 2:
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        elif ch == '+':
            image = image[:, :, :]
        elif ch == '-':
            image = image[::-1]

        if dt is not None:
            image = image.astype(dt)

        return image

    @staticmethod
    def retrieve_name(var):
        callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
        res = [var_name for var_name, var_val in callers_local_vars if var_val is var]
        return ''.join(res)

    # def __repr__(self, check=None, verbose=False):
    #     check = self.image if check is None else check
    #     print("Debugging Starts for:")
    #     name = ''.join(self.retrieve_name(check))
    #     # debug for list that are not arrays
    #     if isinstance(check, list):
    #         if 'numpy' not in str(type(check[0])):
    #             for i, c in enumerate(check):
    #                 name = ''.join(self.retrieve_name(check[i]))
    #                 try:
    #                     print(
    #                         f"{name}:\nShape:\t{np.arrary(c).shape}\nLength:\t{len(c)}\nType:\t{type(c)}\n"
    #                         f"Print (1st value):\n{c[0][0]}\n\n")
    #                 except AttributeError:
    #                     print(c)
    #
    #     elif 'numpy' in str(type(check)):
    #         if not verbose:
    #             print(f"{name}:\t{np.array(check).shape}")
    #             pass
    #         else:
    #             check_type = check.dtype
    #             print(f'Array:\t{name}:\nType:\t{check_type}\nShape:\t{check.shape}\nDims:\t{check.ndim}\n')
    #
    #             try:
    #                 cv2.imshow(name, check)
    #                 # cv2.waitKey()
    #             except cv2.error:
    #                 print('Image visualization failed\n')
    #             try:
    #                 print(
    #                     f'First Value\t{check[0][0] if sum(check[0][0]) != 0 else "Canceled --> PTP: " + str(np.ptp(check))}\n\n')
    #             except (TypeError, IndexError) as e:
    #                 print(f'Cant read image!\nError:\t{e}')
    #                 print(f'{name}:\nType:\t{check_type}\n{check}\n\n')

    # def print_tensor_info(self, tnsr):
    #     name = self.retrieve_name(tnsr)
    #     if not isinstance(tnsr, torch.Tensor):
    #         type_of = type(tnsr)
    #         if 'numpy' not in str(type_of):
    #             tnsr = np.array(tnsr)
    #             print('Type converted to array')
    #             if not np.ndim(tnsr) <= 3:
    #                 try:
    #                     tnsr = tnsr[1:-1]
    #                 except ValueError:
    #                     print(f"{name} type({type_of}) is not a tensor, getting 1st element...")
    #                     tnsr = torch.Tensor(tnsr[0][0])
    #     # Start Printout
    #     print(f"{name.upper()}:\n"
    #           f"SHAPE:\t{tnsr.shape}\n"
    #           f"DTYPE:\t{tnsr.dtype}\n"
    #           f"TYPE:\t{type(tnsr)}\n"
    #           f"MAX:\t{round(tnsr.max(), 2)}\n"
    #           f"MIN:\t{round(tnsr.min(), 2)}\n"
    #           f"MEAN:\t{round(tnsr.mean(), 2)}\n")
