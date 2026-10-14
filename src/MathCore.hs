module MathCore (
    Vector,
    Matrix,
    Label,
    Sample,
    dotProduct,
    euclideanDistance
) where

type Vector = [Double]
type Matrix = [[Double]]
type Label = Int
type Sample = (Label, Vector)

dotProduct :: Vector -> Vector -> Double
dotProduct u v = sum (zipWith (*) u v)

euclideanDistance :: Vector -> Vector -> Double
euclideanDistance u v = sqrt (sum (zipWith (\x y -> (x - y) ^ 2) u v))