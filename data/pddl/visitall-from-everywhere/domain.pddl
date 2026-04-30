(define (domain grid-visit-all)
  (:requirements :strips :negative-preconditions)
  (:predicates
    (place ?x)
    (connected ?x ?y)
    (at-robot ?x)
    (visited ?x)
  )

  (:action move
    :parameters (?curpos ?nextpos)
    :precondition (and
      (place ?curpos)
      (place ?nextpos)
      (at-robot ?curpos)
      (connected ?curpos ?nextpos)
    )
    :effect (and
      (at-robot ?nextpos)
      (not (at-robot ?curpos))
      (visited ?nextpos)
    )
  )
)
