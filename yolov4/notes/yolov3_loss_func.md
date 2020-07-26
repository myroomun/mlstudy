Yolov3 Loss Function (optimized for Yolov4)


What is the output of yolov3
    For each grid cell,
        For each anchors
            confidence,
            probability of classes,
            x, y, w, h

Grid Cell Count: Input Size / Grid Cell Size
Anchor Count: 3
Confidence: 1
Probability: # of classes
Box info: x, y, w, h

Box Calculation
X Y W H are the values in absolute coords
x y w h are the values given by inference
-------------------------------------------------------
tmp x = (sigmoid(x) * scale) - (scale - 1)/2
X = (base x + tmp x) / cols

tmp y = (sigmoid(y) * scale) - (scale - 1)/2
Y = (base y + tmp y) / rows

NOTE that if sigmoid(x) == 0.5, tmp x is -1/2 which is the center point of [base x - 1, base x]
Thus, tmp x is swing over between [base x - 1, base x]

W = exp(w) * Anchor width
H = exp(h) * Anchor width


Loss calculation
BOX Loss: giou loss
Confidence Loss: focal loss = focal factor * cross entropy
    - respond bbox: confidecne answer
    - respond background: (1 - confidence answer) * (max iou < IOU THRESH)
    - focal factor: (respond bbox - confidence predicted)^2
    - cross entropy: (respond bbox + respond background)
Prob Loss: cross entropy(respond bbox, predicted prob) * respond bbox
