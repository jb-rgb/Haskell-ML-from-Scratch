module KNN (
    knnClassify
) where

import MathCore (Sample, Vector, Label, euclideanDistance)
import Data.List (sortBy, sort, group, maximumBy)
import Data.Ord (comparing)

findMode :: [Label] -> Label
findMode [] = error "Cannot find mode of an empty list"
findMode labels =
    let grouped = group (sort labels)
        longest = maximumBy (comparing length) grouped
    in case longest of
        (x:_) -> x
        []    -> error "Empty group encountered"

knnClassify :: Int -> [Sample] -> Vector -> Label
knnClassify k trainData target =
    let
        distances = map (\(lbl, vec) -> (euclideanDistance target vec, lbl)) trainData
        sortedDistances = sortBy (comparing fst) distances
        topKLabels = map snd (take k sortedDistances)
    in
        findMode topKLabels