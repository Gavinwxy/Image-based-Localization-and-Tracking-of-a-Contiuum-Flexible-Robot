import cv2

class ptCoord:
    def __init__(self):
        self.points = (0,0)

    def click_and_mark(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points = (x,y)

def labelPoints(img):
    pt_mark = ptCoord()
    pts = []
    while True:
        cv2.namedWindow('imgLeft')
        cv2.setMouseCallback('imgLeft', pt_mark.click_and_mark)
        cv2.imshow('imgLeft',img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        elif k == ord('a'):
            pts.append(pt_mark.points)
            cv2.circle(img, pt_mark.points, 0, (0, 255, 0), 5)
    cv2.destroyAllWindows()
    return pts