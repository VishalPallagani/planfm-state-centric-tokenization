(define (problem grid-1-2-start-0-0)
  (:domain grid-visit-all)
  (:objects loc-x0-y0 loc-x0-y1)
  (:init
    (at-robot loc-x0-y0)
    (visited loc-x0-y0)
    (connected loc-x0-y0 loc-x0-y1)
    (connected loc-x0-y1 loc-x0-y0)
    (place loc-x0-y0)
    (place loc-x0-y1)
  )
  (:goal
    (and
      (visited loc-x0-y0)
      (visited loc-x0-y1)
    )
  )
)