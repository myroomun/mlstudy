Generalized IOU

A = Answer Area
B = Predicted Area
IOU = (A ∩ B) / (A ∪ B)

Disadvantage
    - If there is no intersected area, IOU is always zero
    - However, consider the below cases
        + First case) IOU = 0, but A and B are close to each other
        + Second case) IOU = 0, and A and B are far from each other
        + The first case should have lower loss than the second case

New IOU = IOU - f(A, B)
Where, f(A, B) is the metric for the distance of A and B

Best candidates for f(A, B)
    - L2: |P0 - P1|^2
    - L1: |P0 - P1|
    - New metric: (Min Convex Hull - (A ∪ B)) / Min Convex Hull


L2/L1 is not good for calculating the geometrical distance
Refer https://giou.stanford.edu
