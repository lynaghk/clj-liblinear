(ns clj-liblinear.test
  (:use [clj-liblinear.core :only [train predict]]))


(let [train-data (concat
                  (repeatedly 300 #(hash-map :class 0 :f {:x (rand), :y (rand)}))
                  (repeatedly 300 #(hash-map :class 1 :f {:x (- (rand)), :y (- (rand))})))
      model (train
             (map :f train-data)
             (map :class train-data)
             :algorithm :l2l2)]
  
  [(predict model {:x (rand) :y (rand)})
   (predict model {:x (- (rand)) :y (- (rand))})])
;;=> [0 1]





(let [train-data (concat
                  (repeatedly 300 #(hash-map :class 0 :f {:x (rand), :y (rand)}))
                  (repeatedly 100 #(hash-map :class 1 :f {:x (- (rand)), :y (- (rand))})))
      model (train
             (map :f train-data)
             (map :class train-data)
             :algorithm :l2l2
             :weights [[0.75 0.25][0 1]]
             )]
  
  [(predict model {:x (rand) :y (rand)})
   (predict model {:x (- (rand)) :y (- (rand))})])
;;=> [0 1]
