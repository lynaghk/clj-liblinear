(ns clj-liblinear.core
  (:refer-clojure :exclude [read-string])
  (:require [clojure.set :refer [union]]
            [clojure.edn :refer [read-string]])
  (:import (de.bwaldvogel.liblinear FeatureNode
                                    Model
                                    Linear
                                    Problem
                                    Parameter
                                    SolverType)))

(set! *warn-on-reflection* true)

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
                                :l1l2_primal SolverType/L1R_L2LOSS_SVC
                                :l1lr SolverType/L1R_LR
                                :l2lr SolverType/L2R_LR)
                    c eps)

        dimensions (dimensions xs)
        xs (into-array (map (fn [instance] (into-array (sort-by #(.index ^FeatureNode %)
                                                               (feature-nodes instance dimensions))))
                            xs))
        ys (into-array Double/TYPE ys)
        prob (new Problem)]

    (set! (.x prob) xs)
    (set! (.y prob) ys)
    (set! (.bias prob) (cond (true? bias) 1
                             (> bias 0) 1
                             :else 0))
    (set! (.l prob) (count xs))
    (set! (.n prob) (count dimensions))

    ;;Train and return the model
    {:liblinear-model (Linear/train prob params)
     :dimensions dimensions}))

(defn predict [model x]
  (Linear/predict (:liblinear-model model)
                  (into-array (feature-nodes x (:dimensions model)))))

(defn save-model
  "Writes the model out to two files specified by the base-file-name which should be a path and base file name. The extention .bin is added to the serialized java model and .edn is added to the clojure dimensions data."
  [model base-file-name]
  (with-open [out-file (clojure.java.io/writer (str base-file-name ".bin"))]
    (Linear/saveModel out-file ^Model (:liblinear-model model)))
  (spit  (str base-file-name ".edn") (:dimensions model)))

(defn load-model
  "Reads a useable model from a pair of files specified by base-file-name. A file with the .bin extension should contain the serialized java model and the .edn file should contain the serialized (edn) clojure dimensions data."
  [base-file-name]
  (let [mdl (Linear/loadModel (clojure.java.io/reader (str base-file-name ".bin")))
        dimensions (read-string (slurp (str base-file-name ".edn")))]
    {:liblinear-model mdl :dimensions dimensions}))
