module MathCore (
    Vector,
    Matrix,
    Label,
    Sample,
    dotProduct,
    euclideanDistance,
    matrixVectorMultiply,
    addVectors,
    relu,
    reluVector
) where

type Vector = [Double]
type Matrix = [[Double]]
type Label = Int
type Sample = (Label, Vector)

dotProduct :: Vector -> Vector -> Double
dotProduct u v = sum (zipWith (*) u v)

euclideanDistance :: Vector -> Vector -> Double
euclideanDistance u v = sqrt (sum (zipWith (\x y -> (x - y) ^ 2) u v))

matrixVectorMultiply :: Matrix -> Vector -> Vector
matrixVectorMultiply matrix vector = map (\row -> dotProduct row vector) matrix

addVectors :: Vector -> Vector -> Vector
addVectors u v = zipWith(+) u v

relu :: Double -> Double
relu x = max 0 x

reluVector :: Vector -> Vector
reluVector v = map relu v