---
layout: page
title: Kannada Posts
---


{% for tag in site.tags %}
  {% if tag[0] == "kannada" %}
  <ul>
    {% for post in tag[1] %}
      <li><a href="{{ post.url }}">{{ post.date | date: "%B %Y" }} - {{ post.title }}</a></li>
    {% endfor %}
  </ul>
  {% endif %}
{% endfor %}
