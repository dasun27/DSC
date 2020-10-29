---
title: "Projects"
permalink: /projects/
layout: posts
entries_layout: grid
author_profile: true
---
{% include base_path %}

{% for post in site.posts reversed %}
  {% include archive-single.html %}
{% endfor %}
