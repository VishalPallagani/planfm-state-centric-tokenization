(define (problem grid-1-1-start-0-0)
  (:domain grid-visit-all)
  (:objects loc-x0-y0)
  (:init
    (at-robot loc-x0-y0)
    (visited loc-x0-y0)
    (place loc-x0-y0)
  )
  (:goal
    (and
      (visited loc-x0-y0)
    )
  )
)