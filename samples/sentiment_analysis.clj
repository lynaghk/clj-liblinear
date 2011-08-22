;;Using clj-liblinear for textual sentiment analysis
;;This is just a demo problem for the sake of example.
;;If you're working on machine learning in Clojure, you're definitely bright enough to solve actual problems.
;;You know, like cancer and shit, so please don't waste your talents on social media marketing. kthx.

(use '[clj-liblinear.core :only [train predict]]
     '[clojure.string :only [split lower-case]])

(def facetweets [{:class 0 :text "grr i am so angry at my iphone"}
                 {:class 0 :text "this new movie is terrible"}
                 {:class 0 :text "disappointed that my maximum attention span is 10 seconds"}
                 {:class 0 :text "damn the weather sucks"}

                 {:class 1 :text "sitting in the park in the sun is awesome"}
                 {:class 1 :text "eating a burrito life is super good"}
                 {:class 1 :text "i love weather like this"}
                 {:class 1 :text "great new album from my favorite band"}])

(let [bags-of-words (map #(-> % :text (split #" ") set) facetweets)
      model         (train bags-of-words (map :class facetweets))]
  
  (map #(predict model (into #{} (split % #" ")))
       ["damn it all to hell!"
        "i love everyone"
        "my iphone is super awesome"
        "the weather is terrible this sucks"]))

;; => (0 1 1 0)
