(define (problem logistics-c2-s2-p4-a1)
  (:domain logistics)
  (:objects
            a0 
            c0 c1 
            t0 t1 
            l0-0 l0-1 l1-0 l1-1 
            p0 p1 p2 p3 
  )
  (:init
    (airplane a0)
    (city c0)
    (city c1)
    (truck t0)
    (truck t1)
    (location l0-0)
    (in-city l0-0 c0)
    (location l0-1)
    (in-city l0-1 c0)
    (location l1-0)
    (in-city l1-0 c1)
    (location l1-1)
    (in-city l1-1 c1)
    (airport l0-0)
    (airport l1-0)
    (package p0)
    (package p1)
    (package p2)
    (package p3)
    (at t0 l0-0)
    (at t1 l1-0)
    (at p0 l0-1)
    (at p1 l1-0)
    (at p2 l1-1)
    (at p3 l1-0)
    (at a0 l1-0)
  )
  (:goal
    (and
        (at p2 l1-1)
        (at p1 l0-0)
    )
  )
)
