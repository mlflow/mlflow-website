---
title: "AI Is Not Creative, Unless You Do This"
description: "An experiment applying the Medici Effect to LLMs shows how cross-pollinating ideas from unrelated fields can significantly boost AI creativity."
slug: ai-not-creative-medici-effect
authors: [khalil-kafrouni]
tags: [ai, creativity, llm, genai, mlflow, prompt-engineering]
thumbnail: /img/blog/ai-creativity-thumbnail.png
image: /img/blog/ai-creativity-thumbnail.png
---

# AI Is Not Creative, Unless You Do This

---

If you're someone with a restless creative itch, you always have this desire to create: to make beautiful art, build software that solves a problem, start a new useful business, do something that pushes the boundaries of humanity even if just a tiny bit. You go ask the AI for some ideas and instead of inspiration, you get generic regurgitated garbage.

Take a look at these answers I got when I asked claude for some YouTube video ideas around tech and AI:

> **1. "I Let AI Run My Life for 7 Days" (Challenge/Vlog format)**  
> Use AI tools (Claude, ChatGPT, various agents) to make every decision for a week — what to eat, how to schedule your day, what to buy, how to respond to emails. Great hook, relatable, easy to make funny or thought-provoking, and naturally showcases multiple AI tools without feeling like an ad.
>
> **2. "The AI Tool Everyone's Sleeping On" (Discovery/Review series)**  
> A recurring format where each episode covers one underrated AI tool or feature people haven't heard of yet — not the obvious ChatGPT/Midjourney stuff, but niche things like AI-powered spreadsheet tools, coding copilots, or research assistants. Series formats build subscriber habits because people come back for the next one.
>
> **3. "Explaining [Complex AI Concept] Using Only Everyday Objects" (Educational/Explainer)**  
> Take something like neural networks, transformers, or how LLMs actually "think," and explain it using physical props, drawings, or analogies — no jargon. This taps into the huge appetite for AI literacy content and can perform really well if the explanation is genuinely clear and visual (think Kurzgesagt-style but for AI).

Now, don't you feel like you've already seen all of these videos? Not only that, if 100 people ask it the same question they are all going to get the same result or highly similar variants of it. This is by design: LLMs, under the hood, are nothing but fancy word predictors, and if their task is to say the most likely sentence, it means they are inherently uncreative. You don't get a creative result by doing what everybody else expects. But here's the good news, there is a solution.

## The Medici Effect

Let's zoom out a little bit and explore how we have solved this problem historically. In 2004, Frans Johansson coined the term "the Medici Effect", which was also the title of his book. In it he argued that human creativity comes from mixing together unrelated fields. He talks about how the Medici family in Florence brought together artists, philosophers, scientists, architects, all to create a melting pot of knowledge which gave birth to the renaissance age.

In other words, the concept is to cross-pollinate one idea with another to create something new, forcing the creativity to come from the mashup of ideas rather than from naturally thinking of something novel. So, could we apply it to LLMs?

## Let's Run an Experiment

We will simulate a bunch of people coming to the AI seeking ideas on a variety of topics: launching a startup, writing an article, making an art installation, doing research, building an open-source tool, anything that has to do with creativity. We will split the simulations into 3 groups:

- A control group which will use a regular, direct approach where we simply ask the LLM for ideas
- A placebo group where we add a sentence to the prompt to make the AI think outside the box and be creative
- And a treatment group which will use randomized cross-pollination.

For this, we will create long list of concepts and notions from a variety of fields. We will ask Claude to generate 20 concepts from each of the categories: math, economics, psychology, biology, architecture, physics, and computer science.

Here is the exact prompt I used along with a few examples:

> **Prompt:**
>
> These are our focus domains: math, economics, psychology, biology, architecture, physics, and computer science. In each of them, find 20 concepts/ideas and write their titles and descriptions. Make the description a couple sentences rather than just a mere headline. I want it to be so that if somebody read it who has no idea what that concept is, they would understand it. Make sure your output is in JSON.

**Examples Concepts:**

```json
 "economics": [
    {
      "title": "Opportunity Cost",
      "description": "This is the idea that every choice you make means giving up the next-best alternative you didn't choose. For example, if you spend an evening watching a movie, the opportunity cost is whatever else you could have done with that time, like studying or earning money, and this concept helps explain why every decision, even free ones, has a hidden 'cost'."
    },
    {
      "title": "Comparative Advantage",
      "description": "This principle explains why trade between two people, companies, or countries can benefit both sides, even if one of them is better at producing everything. It works because each party should focus on producing whatever they give up the least to make, and then trade for the rest, so total production and wealth increase for everyone involved."
    },
    {
      "title": "Moral Hazard",
      "description": "This describes a situation where someone takes on more risk because they know someone else will bear the consequences if things go wrong. A classic example is a person driving more carelessly after getting car insurance, since the financial cost of an accident is now covered by the insurance company rather than themselves."
    }
  ]
```

Then, we will create another AI evaluator that judges the LLM's output in MLflow. It will give each idea a score from 1 to 10 on innovation, feasibility, and significance. Finally, we will compare the results of the groups. The LLM used for both the execution and the evaluation is Deepseek v4 flash.

## Evaluating the Traces

I ran the agents and traced everything in MLflow. Each of the three groups got over 500 runs, everything was stored. Then, I built a scorer agent in MLflow that ran and graded each and every idea on its innovation, feasibility, and significance (how useful or transformative the idea is.) Then, to end up with one final score, I took the product of all three:

$ \text\{Score\} = \frac\{I \times F \times S\}\{1000\} $

The reason for selecting this grading rubric is that if either one of the three metrics was too low, I want it to drag down the entire score with it. So, if an idea has 10 innovation and 10 significance but 0 feasibility, it would count for zero.

## Score Extremization

Before going into the results and their analysis, there was a problem that I encountered that I had to solve: when you use an LLM to score some input, it generally tends to have a rather concentrated grade around the mean. This is more strongly the case when the input itself is generated by LLMs and has therefore a certain threshold of quality. In our case, in the first pass, I had a very high concentration of the grades around 5/10 on each metric with a standard deviation of 0.1, making it nearly impossible to clearly see a difference between our three groups.

There are several ways to solve this issue, some of them being statistical techniques that could be applied to the grades, but in our case, it was simply resolved by giving the evaluator clear guidelines to follow in its system prompt for grading.

```prompt
1-2 => genuinely bad
3-4 => mediocre, forgettable
5-6 => decent, worth a nod
7-8 => genuinely good, made you sit up
9-10 => exceptional, you'll remember it
```

## Results

![](/img/blog/ai-creativity-results.png)

Lo and behold, after over 500+ requests for ideas in each of the three groups, here are the average scores:

| Control | Placebo | Experiment |
| ------- | ------- | ---------- |
| 0.21    | 0.20    | **0.26**   |

The experiment ideas scored an average of 0.26, a 23% increase from the control group, with a p-value < 0.0001. Meaning that the ideas generated with the concept cross-pollination were significantly better on the combined score of innovation, feasibility, and significance. However, and here is the more interesting part, on the individual metrics, it was not a clear winner. You can clearly see this in this table of the average metric scores per group.

| Metric       | Control  | Placebo  | Experiment |
| ------------ | -------- | -------- | ---------- |
| Innovation   | 5.57     | **7.28** | 6.35       |
| Feasibility  | **6.26** | 4.70     | 6.25       |
| Significance | 5.92     | 5.75     | **6.29**   |

In fact, if we measure innovation alone, the placebo group was by far the highest. This is the group where we specifically asked the LLM to be creative, break norms, and think outside the box. The result was that it generated novel ideas that were too ambitious to ever be realized.

On the other hand, what the cross-pollination group (experiment) results show is that this technique makes it generate more innovative results that are balanced on all 3 metrics.

For a more hands-on view, I will present to you a sample idea request made in each of the three groups along with their respective outputs. In this sample, the goal was to get ideas for a youtube video about education:

> **1. Control: simply asking for the idea**
>
> A teacher enters a classroom where every desk has a single, seemingly random object (a rubber chicken, a bicycle pump, a framed photo of a cat). Instead of a lesson plan, students must invent a 1-minute TED talk that connects their object to a core academic concept like gravity or democracy. The video documents the chaotic, brilliant ideas that emerge—and the real learning that happens when forced to bridge the absurd and the academic.
>
> **2. Placebo: asking the AI to think creatively and outside the box**
>
> A video where the viewer is forcibly enrolled in a university that exists only inside a livestreamed video game, and each wrong answer or skipped assignment causes a real, randomized consequence for the host—like deleting a cherished memory or destroying a physical object—blurring the line between performance, punishment, and pedagogy.
>
> **3. Experiment: cross-pollinating with the concept of annealing (heating a metal then cooling it slowly to redistribute the defects)**
>
> Sit a student down with a really hard problem. Instead of solving it, have them deliberately record the worst possible approach, then a slightly less bad one, then a mediocre one, and so on, gradually refining their thinking until they arrive at a strong solution. The video documents this slow "cooling down" from terrible to great, showing how allowing yourself to explore awful ideas ultimately produces a more robust, less brittle understanding than trying to be clever right away.

## Conclusion

This experiment reveals something fascinating: just like we become more creative by constraining our thought process and finding inspirational links from other domains, so do LLMs. Even though they are word predictors, forcing them to find distant links can indeed yield inspiring results. So, the next time you are out there seeking some exciting ideas from your favorite LLM, maybe try forcing it to look into fields that have nothing to do with it.

If you want to improve your LLM apps and push their boundaries, MLflow is your playground. Star us on [Github](https://github.com/mlflow/mlflow)!
