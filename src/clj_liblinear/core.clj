(ns clj-liblinear.core
  (:use [clojure.set :only [union]])
  (:import (de.bwaldvogel.liblinear FeatureNode
                                    Linear
                                    Problem
                                    Parameter
                                    SolverType)))

(defmacro set-all! [obj m]
  `(do ~@(map (fn [[k# v#]]
                `(set! (. ~obj ~k#) ~v#))
              m)
       ~obj))

(defn feature-nodes [x dimensions]
  (cond
   (map? x) (for [[k v] x] (FeatureNode. (k dimensions) v))
   (set? x) (for [v x :when (dimensions v)] (FeatureNode. (dimensions v ) 1))))

(defn dimensions
  "Get all of the dimensions in a collection of map/set instances, return a map of dimension -> index"
  [xs]
  (let [dimnames (cond (every? map? xs) (into #{} (flatten (map keys xs)))
                       (every? set? xs) (apply union xs))]
    (into {} (map vector dimnames (range 1 (inc (count dimnames)))))))

(defn train
  "Train a LIBLINEAR model on a collection of maps or sets, xs, and a collection of their integer classes, ys."
  [xs ys & {:keys [c eps algorithm bias]
                      :or {c 1, eps 0.1, algorithm :l2l2, bias 0}}]
  (let [params (new Parameter (condp = algorithm
                                  :l2lr_primal SolverType/L2R_LR
                                  :l2l2 SolverType/L2R_L2LOSS_SVC_DUAL
                                  :l2l2_primal SolverType/L2R_L2LOSS_SVC
                                  :l2l1 SolverType/L2R_L1LOSS_SVC_DUAL
                                  :multi SolverType/MCSVM_CS
                                  :l1l2 SolverType/L1R_L2LOSS_SVC
                                  :l1lr SolverType/L1R_LR
                                  :l2lr SolverType/L2R_LR)
                    c eps)
        
        dimensions (dimensions xs)
        xs (into-array (map (fn [instance] (into-array (sort-by #(.index %)
                                                               (feature-nodes instance dimensions))))
                            xs))
        ys (into-array Integer/TYPE ys)
        prob (new Problem)]
    
    (set-all! prob {:x xs
                    :y ys
                    ;;Waldvogel's port will add a free dimension when bias > 0; allow our users to more explicitly specify true/false.
                    :bias (cond (true? bias) 1
                                (> bias 0) 1
                                :else 0)
                    :l (count xs)
                    :n (count dimensions)})
    
    ;;Train and return the model
    {:liblinear-model (Linear/train prob params)
     :dimensions dimensions}))

(defn predict [model x]
  (Linear/predict (:liblinear-model model)
                          (into-array (feature-nodes x (:dimensions model)))))
