Tuning:  Plain at least 4 different tune models

Freezing 
M1 – train only the head (baseline model – this should already be done, don’t touch these primers)
M2 – Un freezes the beginning 1-3
M3 – unfreezes the middle 4-6 ( beginning is now frozen again)
M4 – unfreezes the End 7-8 ( Middle is frozen again)
M5 – fu Freez the interior backbone

Rotating opinions to play around with
O1 – ±10° 
O2 – ±20°
 O3 – ±30° 
O4 – ±45°

Learning rate opinions ( Would like to keep a low LR, don’t go over 4)
O1 – 5
O2 – 6
O3 -4

Color Opinions
O1 – Brightness (±10%) 
O2 – Brightness + Contrast (±15%) 
O3 – Brightness + Contrast + Saturation (±20%) 
O4 – Full color jitter (brightness, contrast, saturation, hue)

Batch size Opinions
O1 – 16
 O2 – 24 
O3 – 32 
O4 – 36 (max)

Dropout opinions
O1 – 0.2
 O2 – 0.3 
O3 – 0.5 
O4 – 0.6 (strong regularization)

augmentation strengths opinions
O1 – Light (flip + small rotation only) 
O2 – Medium (rotation + color jitter) 
O3 – Strong (rotation + color + zoom + shift) 
O4 – Very Strong (all augmentations maxed)

Zoom Options 
O1 – 0.9–1.1 (very light zoom) 
O2 – 0.8–1.2 
O3 – 0.7–1.3 
O4 – 0.6–1.4 (aggressive)

Shift Options 
O1 – ±5% 
O2 – ±10% 
O3 – ±15% 
O4 – ±20%

Flip Options 
O1 – Horizontal only 
O2 – Horizontal + Vertical
 O3 – Vertical only 
O4 – No flip
