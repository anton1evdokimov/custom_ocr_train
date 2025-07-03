import os

import cv2
# p = 'images/vlcsnap-2025-05-16-08h49m47s436_7.png'
for i in os.listdir('images'):
    if i.endswith('.png'):
        img = cv2.imread(os.path.join("images", i))
        print(img.shape)
# p = 'result_png'
# RENAME = False

# if RENAME:
#     for i in os.listdir(p):
#         path = os.path.join(p, i)
#         os.rename(path, path.replace(".jpg_", "_"))
#     for i in os.listdir('result_png'):
#         print(i)

# WRITE = True
# if WRITE:
#     res = ["filenames,words\n"]

#     res.extend(map(lambda x: f"{x},\n", filter(lambda x: x.endswith('.png') ,os.listdir('result_png'))))
#     # with open("label.txt", 'w') as file:

#     res.sort()

#     with open("label.txt", 'w') as file:
#         # file.write("filenames,words\n")
#         for i in res:
#             file.write(i)
