"""Shared on-video drawing primitives (outlined text, translucent label blocks, HUD panels)."""
import cv2

FONT        = cv2.FONT_HERSHEY_DUPLEX
LABEL_SCALE = 0.62
LABEL_THICK = 2


def txt(img, text, x, y, scale, color, thick, font=FONT):
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def label_block(img, lines, colors, cx, y_start, bg_color, scale=LABEL_SCALE, thick=LABEL_THICK, alpha=0.75):
    pad_x, pad_y, gap = 12, 6, 4
    sizes = [cv2.getTextSize(l, FONT, scale, thick)[0] for l in lines]
    max_w = max(s[0] for s in sizes)
    total_h = sum(s[1] for s in sizes) + gap*(len(lines)-1)
    h_img, w_img = img.shape[:2]
    bx1 = max(0, cx - max_w//2 - pad_x); by1 = max(0, y_start - pad_y)
    bx2 = min(w_img-1, cx + max_w//2 + pad_x); by2 = min(h_img-1, y_start + total_h + pad_y)
    ov = img.copy()
    cv2.rectangle(ov, (bx1, by1), (bx2, by2), bg_color, -1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)
    y_cur = y_start
    for line, color, (tw, th) in zip(lines, colors, sizes):
        txt(img, line, cx - tw//2, y_cur+th, scale, color, thick)
        y_cur += th + gap

