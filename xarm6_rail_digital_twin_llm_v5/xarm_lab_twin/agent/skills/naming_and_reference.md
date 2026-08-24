---
name: naming-and-reference
audience: intake
applies_to: any instruction that names, describes or points at an object
---

# The operator's words become bindings

Downstream of you, every phrase in the task text is matched against the
scene's object names and aliases, and whatever it matches is pulled into the
task: measured, reported to the planner, and in some cases used to fit
equipment. That matching is deliberately eager, because failing to notice
that an operator meant a real object is worse than noticing one too many.

The consequence is the thing to internalise: **a word does not have to be
about an object to bind to one.** Scene-setting, throwaway locatives and
casual class nouns bind exactly as strongly as a deliberate reference. You
are the layer that decides which words survive into the task text, so you
are the layer that controls what binds.

## Keep every reference the operator gave you, and add none

You have not seen the scene. You do not know how many blue cubes are in it,
so "the blue cube" is not yours to question: it is an ordinary reference,
and the layers below you resolve it against the real object list and report
a genuine collision when there is one. Asking the operator to identify an
object you simply cannot count is noise, and it stops a run that had nothing
wrong with it.

Pass such a reference through in the operator's own words. Where the
operator has already given an exact name, keep that exact name -- do not
paraphrase it back into a description, because the name resolves cleanly and
the description may not.

The ambiguity that IS yours is ambiguity visible in the sentence: an
instruction that contradicts itself, one that could describe two different
goals, one where the operator has clearly left a word out. Judge the
instruction, never the scene.

## Let nothing that is not acted on read as an object

This is the half that gets forgotten. An instruction like "put it back where it
started, on the bench" contains one object reference and one piece of scene
decoration. The decoration is not harmless: "on the bench" is most of the
alias of a piece of labware that sits on the bench, so it binds to that
labware, which is then measured and announced to the planner as if the
operator had asked for it. Nothing downstream can tell the difference,
because by then the operator's phrasing is gone and only the binding remains.

Rewrite so that every phrase either names something the task acts on, or
cannot be read as naming anything. "Put it back where it started" says the
same thing and binds nothing.

## Never mention a class the task does not involve

Class nouns -- plate, well plate, tip box, tip rack, tube, rack -- are read
downstream as statements about what is being handled, and some of them cause
equipment to be fitted before the planner ever runs. A task about blocks that
happens to mention a plate is a task that gets a plate handled. Mention a
class when the task handles it. Otherwise leave it out, including from
constraints and from the restatement of the goal.

The same applies to effectors. Which effector is fitted follows from the
class of object being handled, so name the class and let that follow. Do not
name an effector the task has no reason to change.

## Reference to an earlier state is not reference to an object

"Back to its original position", "where it was", "how it started" point at a
remembered state, not at a thing in the scene. Keep them as goal statements
in exactly those terms. Do not try to help by adding where you imagine that
was -- you have not seen the scene, and a location you supply is an invention
that later layers will treat as measured fact.

## What to write

Restate the goal using explicit names for the objects acted on, no names for
anything else, no class noun the task does not handle, and no location you
were not given. When an instance is genuinely ambiguous, ask instead of
picking -- and ask about the object, naming the candidates you think are
meant.
