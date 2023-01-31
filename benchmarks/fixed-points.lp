% Compute the fixed points (stable states) of a model
% Principle: enumerate all states and keep those with no playable local transition
% Usage: see REAMDE.txt
%
% Glossary:
% Fixed point = Stable state
%             = state with no successor state
%             = state with no playable local transition
% State = a set of local states (automaton name & value for each automaton)



%%% Initialization (extended facts)
% Automata names
automaton(Automaton) :- automaton_level(Automaton, _).
% Local transition names
local_transition(Transition) :- target(Transition, _, _).

%%% Fixed points enumeration

% Enumerate all possible states (one local state per automaton)
1 { local_state(Automaton, Level) : automaton_level(Automaton, Level) } 1 :-
  automaton(Automaton).

% Compute not playable transitions in the current state
unplayable(Transition) :-
  local_state(Automaton, LevelI),
  condition(Transition, Automaton, LevelJ),
  LevelI != LevelJ.

% Constraint: discard states with a playable transition
:- not unplayable(Transition), local_transition(Transition).



#show local_state/2.

