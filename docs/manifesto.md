# A Note from the Developer

**Ryan McCann**

When we started building FUSION, I didn't expect it to become a multi-year project—let alone something we'd care about enough to shape our research around. By the time I'm writing this, we've been working on FUSION for about three and a half years. This page is my attempt to explain why FUSION looks the way it does, how it got here, and where we're trying to take it.

## Why FUSION exists

Back in 2022, we decided to build an optical network simulator in Python. The choice wasn't accidental: Python gives us a natural path to AI and machine learning workflows, and it lowers the barrier for contributors who want to extend the simulator without fighting tooling or language friction.

What might surprise people is that FUSION was closed source for its first several versions. Early on, it was built primarily for our lab's internal research: move fast, test ideas, publish, repeat. That approach is common in research software—and honestly, it works for producing papers.

But it comes with a cost: the moment you stop maintaining the code, the work becomes hard to reproduce, hard to extend, and hard for the community to trust.

We didn't want FUSION to die at graduation.

## The shift to open source (and what changed in v6)

By version 5, we released what we had. Version 6 was our first serious attempt to make FUSION usable beyond our immediate needs.

That meant confronting feedback we had been hearing consistently:

- "I can't tell where to start."
- "I don't know what's current versus legacy."
- "I don't know what FUSION supports without digging through the code."
- "I don't know how to do basic tasks without tribal knowledge."

Version 6 is our line in the sand: this is where FUSION becomes a project meant for other people—not just for us.

And yes, that shift comes with tradeoffs. Building an open-source simulator takes time that could have gone into producing more papers. We chose the harder path anyway because we believe the optical networking community needs something better than a scattered set of unmaintained, hard-to-run research artifacts.

## What we're building toward

Our goal is simple to say and hard to execute:

**We want FUSION to be a trusted baseline for optical network simulation.**

Not *the only* simulator. Not a monopoly. Not a "winner." A baseline: a simulator people can rely on, extend, compare against, and build research on top of without starting from scratch every time.

If your goal is to publish as fast as possible and never think about maintainability, documentation, or reuse, FUSION will probably feel like extra friction. That's okay. FUSION is for people who care about research *and* software quality—because long-term progress depends on both.

## How to think about the codebase

FUSION has history. It includes newer architecture and older paths that existed before we had a mature documentation system or consistent conventions. We're actively moving toward clearer structure and a more predictable user experience, but we also refuse to do the common research-software move of rewriting everything at once and breaking everyone's workflow.

So the operating principle is:

- modernize iteratively,
- keep the simulator usable,
- and make the recommended path obvious.

If you ever find yourself wondering "what's the intended way to do this today?", that's a documentation bug—we want to know.

## What you can expect from us

FUSION is maintained by a small team under the guidance of our principal investigator. That means you'll sometimes see rough edges, inconsistencies, or transitions in progress. It also means we take feedback seriously, because we can't improve what we don't see.

Here's what we *will* do:

- Treat documentation as a first-class feature.
- Make it easier to understand the architecture and the workflow.
- Help users get unstuck through GitHub issues and community support.
- Keep pushing toward reproducibility, clarity, and extensibility.

Over time, we also plan to invest in the things that make open-source projects scalable: onboarding materials, online events, community channels, version kickoffs, and tutorial content.

## What we ask from contributors

If you want to contribute to FUSION, the expectation is that we build like engineers, not just like researchers:

- clear interfaces,
- readable code,
- tests where they matter,
- documentation that matches behavior,
- and changes that a new person can understand without a private tour.

We're not trying to police anyone's style or "take shots" at researchers—research incentives are real. But we're intentionally building an ecosystem, not a one-off prototype. That requires discipline.

## How we see other simulators

We've looked at essentially every optical networking simulator we could find. We don't see them as enemies. We see them as different design choices—often clever ones—with different goals, constraints, and tradeoffs.

We respect other maintainers, and we're open to collaboration. Open source works when people treat it as a shared foundation, not a zero-sum contest.

## An invitation

If you're new here, don't be intimidated. Ask questions. Open issues. Tell us where you got stuck. We've already met with users directly to help them ramp up, and we're happy to keep doing that when we can.

If you want to improve something, we want to work with you. If you want to build research on top of FUSION, we want you to succeed. The point is not that FUSION is "finished." The point is that it is *alive*, maintainable, and built to outlast any single paper or any single student.

Thanks for finding the project. We're glad you're here.
