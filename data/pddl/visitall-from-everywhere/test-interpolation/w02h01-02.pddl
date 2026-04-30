(define (problem grid-2-1-start-1-0)
  (:domain grid-visit-all)
  (:objects loc-x0-y0 loc-x1-y0)
  (:init
    (at-robot loc-x1-y0)
    (visited loc-x1-y0)
    (connected loc-x0-y0 loc-x1-y0)
    (connected loc-x1-y0 loc-x0-y0)
    (place loc-x0-y0)
    (place loc-x1-y0)
  )
  (:goal
    (and
      (visited loc-x0-y0)
      (visited loc-x1-y0)
    )
  )
)