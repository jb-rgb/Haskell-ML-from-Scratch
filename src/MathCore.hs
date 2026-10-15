module MathCore (
    Vector,
    Matrix,
    Label,
    Sample,
    dotProduct,
    euclideanDistance,
    matrixVectorMultiply,
    addVectors,
    subtractVectors,
    relu,
    reluVector,
    oneHotEncode,
    transposeMatrix,
    reluDerivativeVector,
    scalarMultiply,
    scalarMultiplyMatrix,
    outerProduct
) where

import Data.List (transpose)

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

subtractVectors :: Vector -> Vector -> Vector
subtractVectors u v = zipWith (-) u v

oneHotEncode :: Int -> Label -> Vector
oneHotEncode size label =
    [if i == label then 1.0 else 0.0 | i <- [0..size - 1]]

transposeMatrix :: Matrix -> Matrix
transposeMatrix = transpose

reluDerivative :: Double -> Double
reluDerivative x = if x > 0 then 1.0 else 0.0

reluDerivativeVector :: Vector -> Vector
reluDerivativeVector v = map reluDerivative v

scalarMultiply :: Double -> Vector -> Vector
scalarMultiply scalar v = map (* scalar) v

scalarMultiplyMatrix :: Double -> Matrix -> Matrix
scalarMultiplyMatrix scalar m = map (scalarMultiply scalar) m

outerProduct :: Vector -> Vector -> Matrix
outerProduct u v = map (\x -> map (* x) v) u