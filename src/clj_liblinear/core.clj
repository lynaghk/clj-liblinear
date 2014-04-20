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
   (map? x) (for [[k v] x :when (contains? dimensions k)] (FeatureNode. (get dimensions k) v))
   (set? x) (for [v x :when (dimensions v)] (FeatureNode. (get dimensions v) 1))))

(defn dimensions
  "Get all of the dimensions in a collection of map/set instances, return a map of dimension -> index"
  [xs]
  (let [dimnames (cond (every? map? xs) (into #{} (flatten (map keys xs)))
                       (every? set? xs) (apply union xs))]
    (into {} (map vector dimnames (range 1 (inc (count dimnames)))))))

(defn- bias-feature [dims] (FeatureNode. (inc (count dims)) 1))

(defn- feature-array
  "Features are sorted by index. If bias is active, an extra feature is added."
  [bias dims instance]
  (let [nodes (sort-by #(.index ^FeatureNode %) (feature-nodes instance dims))]
    (if (pos? bias)
      (into-array (concat nodes [(bias-feature dims)]))
      (into-array nodes))))

(defn- correct-predictions
  [target labels]
  (count (filter true? (map = target labels))))

(defn train
  "Train a LIBLINEAR model on a collection of maps or sets, xs, and a collection of their integer classes, ys."
  [xs ys & {:keys [c eps algorithm bias cross-fold]
                      :or {c 1, eps 0.1, algorithm :l2l2, bias 0, cross-fold nil}}]
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
        bias       (if (or (true? bias) (pos? bias)) 1 0)
        dimensions (dimensions xs)
        xs         (into-array (map #(feature-array bias dimensions %) xs))
        ys         (into-array Double/TYPE ys)
        prob       (new Problem)]

    (set! (.x prob) xs)
    (set! (.y prob) ys)
    (set! (.bias prob) bias)
    (set! (.l prob) (count xs))
    (set! (.n prob) (+ (count dimensions) bias))

    ;;Train and return the model
    {:target          (when cross-fold 
                        (let [target (make-array Double/TYPE (count ys))]
                          (Linear/crossValidation prob params cross-fold target)
                          (println (format "Cross Validation Accuracy = %g%%%n"
                                     (* 100.0 (/ (correct-predictions target ys) (count target)))))
                          target))
     :liblinear-model (Linear/train prob params)
     :dimensions dimensions}))

(defn predict [model x]
  (let [m ^Model (:liblinear-model model)]
    (Linear/predict m (feature-array (.getBias m) (:dimensions model) x))))

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

(defn get-coefficients
  "Get the nonzero coefficients of a given model, represented as a map from feature name to coefficient value. The intercept corresponds to the (constant) feature named :bias."
  [model]
  (let [ ;; Check if the model contains a bias coefficient (intercept).
        include-bias (<= 0
                         (.getBias ^de.bwaldvogel.liblinear.Model (:liblinear-model model)))
        ;; Get a vector of the coefficients (ordered as in the
        ;; internal liblinear representation.
        coefficients-vector (-> model
                                :liblinear-model
                                (#(.getFeatureWeights
                                   ^de.bwaldvogel.liblinear.Model %))
                                vec)
        ;; Get the indices (in the above ordering) corresponding to
        ;; the various feature names.
        feature-indices (if include-bias
                          (assoc (:dimensions model)
                            ;; The bias feature is always the last one.
                            :bias (count coefficients-vector))
                          (:dimensions model))]
    ;; Create a hashmap containing the coefficients by name.
    (into {}
          (for [[feature-name feature-index] feature-indices
                :let [coefficient (coefficients-vector
                                   ;; dec, to start from 0, not 1
                                   (dec feature-index))]
                :when (not (zero? coefficient))]
            [feature-name coefficient]))))
