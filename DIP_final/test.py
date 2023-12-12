def reduce_effect(img):
    h, w, n = img.shape
    cx = w / 2
    cy = h / 2
    radius = 100
    r = int(radius / 2.0)
    compress = 8
    new_img = img.copy()
    for i in range(w):
        for j in range(h):
            tx = i - cx
            ty = j - cy
            x = int(cx + (math.sqrt(math.sqrt(tx * tx + ty * ty)) * compress * math.cos(math.atan2(ty, tx))))
            y = int(cy + (math.sqrt(math.sqrt(tx * tx + ty * ty)) * compress * math.sin(math.atan2(ty, tx))))
            if x < 0 and x > w:
                x = 0
            if y < 0 and y > h:
                y = 0
            if x < w and y < h:
                new_img[j, i, 0] = img[y, x, 0]
                new_img[j, i, 1] = img[y, x, 1]
                new_img[j, i, 2] = img[y, x, 2]
    return new_img

if __name__ == "__main__":
    img = cv2.imread("4.jpg")
    enlarge_img = enlarge_effect(img)
    frame = reduce_effect(img)
    cv2.imshow("1", img)
    cv2.imshow("2", enlarge_img)
    cv2.imshow("3", frame)
    cv2.waitKey()
    cv2.destroyAllWindows()
