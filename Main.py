"""
Program loop over images in folder match 2 templates to calculate their distance between them.

By:
Tal A. Toledano
"""

# IMPORTS
from config import *
from cls.Matchers import Template
from cls.Preprocess import Prep

# CONSTANTS AND FLAGS
plt.rcParams["figure.figsize"] = [20, 10]
plt.rcParams["figure.autolayout"] = True
verbose = True
show = False


def main():
    # Reading files in lookup folder
    rerun = False
    lookup_folder = os.path.join(os.getcwd(), 'data', f'{FOLDER}')
    path = get_path(lookup_folder, '')
    files_list = os.listdir(path)
    tmt_files = sorted([name for name in files_list if name.startswith('temp')])
    tmt_paths = [get_path(lookup_folder, name) for name in tmt_files]

    # Reading Templates
    tmt = Template(tmt_paths)
    print(tmt)
    t2, t1 = tmt.template

    # Reading Images
    img_files = [f for f in files_list if f not in tmt_files and f.endswith(tuple(ext))]
    img_paths = [get_path(lookup_folder, name) for name in img_files]
    n_image = len(img_paths)
    print(f'Images in Library:\t{n_image}\n\n')

    # Loop through images for matches
    img_dict = dict()
    for ifile in img_paths:
        # Preprocess Image
        name = ifile.split("data")[-1][3:]
        p = Prep(ifile)
        image = p.origin
        gray = p.preprocess(**prep_param)

        # INFO printout
        if verbose and not rerun:
            p.get_info(image, show)
            p.get_info(t1, show)
            p.get_info(t2, show)
            rerun = True
        print(f'############\n\nReading Image:\nNAME:\t{name}')

        # Run Matcher
        bbox1, dict1 = tmt.run_template(t1, gray)
        (xmin, ymin), (xmax, ymax) = bbox1
        (mX, mY) = tmt.get_bbox_mp(bbox1)

        gray[ymin:ymax, xmin:xmax] = 0  # optimize by masking 1st match
        bbox2, dict2 = tmt.run_template(t2, gray)
        (mX2, mY2) = tmt.get_bbox_mp(bbox2)

        # Calc Templates Distance
        mp1, mp2 = (mX, mY), (mX2, mY2)
        dist = tmt.get_dist(mp1, mp2)

        # Add Plot Details
        img_dict[name] = p.origin[:, :, ::-1], dist, p.origin.shape
        cv2.rectangle(p.origin, *bbox1, G, 2)
        cv2.rectangle(p.origin, *bbox2, B, 2)
        cv2.circle(p.origin, mp1, 5, G, -1)
        cv2.circle(p.origin, mp2, 5, B, -1)
        cv2.line(p.origin, (mX, mY), (mX2, mY2), color=R)
        cv2.putText(p.origin, f"{dist:.1f}", tmt.get_bbox_mp((np.subtract(mp2, (100, 10)), mp1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, B, 2)

    # Visualize
    count = 0
    prev = None
    for k, v in img_dict.items():
        if count != 0:
            delta = round(img_dict[k][1] - img_dict[prev][1], 3)
            stat = 'Expand' if delta > 0 else 'Retract'
            figure, ax = plt.subplots(1, 2, figsize=(20, 10))
            ax[0].imshow(img_dict[prev][0])
            ax[1].imshow(img_dict[k][0])
            ax[0].set(title=prev, aspect=1, xticks=[], yticks=[])
            ax[1].set(title=k, aspect=1, xticks=[], yticks=[])
            plt.suptitle(f"Delta:  {delta} ({stat.title()})")
            plt.show()
        prev = k
        count += 1


if __name__ == '__main__':
    main()
