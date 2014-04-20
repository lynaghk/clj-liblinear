(ns clj-liblinear.test
  (:use [clj-liblinear.core :only [train predict get-coefficients reset-random]]
        clojure.test)
  (:import java.util.Random))


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


(defn inverse-logit [x]
  (/ (inc (Math/exp (- x)))))


(defn generate-logistic-observations
  "Generate n pseudorandom observations by a logistic model.
The feature values are iid samples of standard normal distribution.
The model coefficients are specified as a map from feature name to coefficient value.
The intercept is specified in feature name :intercept."
  [n coefficients seed]
  (let [;; To make the test consistent, we do not use clojure.core's
        ;; usual rand function here.
        ;; One might consider using org.clojure/data.generators for such
        ;; needs in the future.
        prng (Random. seed)
        rand-normal #(.nextGaussian ^Random prng)
        rand-uniform #(.nextDouble ^Random prng)
        feature-names (keys (dissoc coefficients
                                    :intercept))
        intercept (or (:intercept coefficients)
                      0)]
    (repeatedly n
                (fn []
                  (let [features (into {}
                                       (for [feature-name feature-names]
                                         [feature-name (rand-normal)]))
                        prob (inverse-logit
                              (reduce +
                                      (cons intercept
                                            (for [[feature-name feature-value] features]
                                              (* (coefficients feature-name)
                                                 feature-value)))))
                        observed-class (Math/signum (- prob
                                                       (rand-uniform)))]
                    {:f features
                     :class observed-class})))))


(deftest regression
  (let [;; Prepare training data
        train-data (generate-logistic-observations 400
                                                   {:x -2
                                                    :y 1
                                                    :intercept 1}
                                                   0)
        ;; Reset liblinear's PRNG
        _ (reset-random)
        ;; Train model
        model (train
               (map :f train-data)
               (map :class train-data)
               :algorithm :l1lr
               :c 10
               :bias true)]
    (is (= (get-coefficients model)
           {:intercept 0.7628104793079575,
            :y 0.8902534641791852,
            :x -1.9527156975645894}))))
