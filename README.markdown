                                         _ __  
                                        | '_ \ 
                                        | | | |
      /                             \   |_| |_| 
     /                               \        
     |      +             /          |        
     |       +     +     /           |      
     |                  /   -        |      ____  _   _         _  _  _      _  _                            
     |     +    +   +  /  -          |     / ___|| | (_)       | |(_)| |__  | |(_) _ __    ___   __ _  _ __  
     |                /        -     |    | |    | | | | _____ | || || '_ \ | || || '_ \  / _ \ / _` || '__| 
     |     +   +     / -   -         |    | |___ | | | ||_____|| || || |_) || || || | | ||  __/| (_| || |    
     |              /                |     \____||_|_/ |       |_||_||_.__/ |_||_||_| |_| \___| \__,_||_|    
     |    +        /   -    -        |             |__/                                                      
     |          + /      -           |    
     \           /   -               /    
      \                             /                                                           


This is a Clojure wrapper around Benedikt Waldvogel's [Java port](http://www.bwaldvogel.de/liblinear-java) of [LIBLINEAR](http://www.csie.ntu.edu.tw/~cjlin/liblinear/), a linear classifier that can handle problems with millions of instances and features.
Essentially, it is a support vector machine optimized for classes that can be separated without projecting into some fancy-pants kernel space.

## Install

Add

```clojure
[clj-liblinear "0.1.0"]
```

to the `:dependencies` vector in your `projects.clj` file.


## Examples

Clj-liblinear takes maps as instances:

```clojure
(use '[clj-liblinear.core :only [train predict]])
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
```

If you are concerned only with occurrences (rather than continuous variables), you can use sets.
These will be expanded into indicator variables for classification.
For instance, you can easily do simple text classification based on word occurrence:

```clojure
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
```



You can also pass weights to the train method, for example if the major class had 100k observations and the minor class had 10k observations, you might want to call:
```clojure
(train bags-of-words (map :class facetweets) :weights [[0.1 0.9][ 1 0]])
```


## Thanks
The National Taiwan University Machine Learning Group for [LIBLINEAR](http://www.csie.ntu.edu.tw/~cjlin/liblinear/), and Benedikt Waldvogel his [Java transliteration](http://www.bwaldvogel.de/liblinear-java).

This project is sponsored by [Keming Labs](http://keminglabs.com), a technical design studio specializing in data visualization.
