---
layout: page
title: Nastavnici
description: Pregled svih nastavnika na predmetu.
---

# Nastavnici
## Nositelji

{% assign leads = site.staffers | where: 'leads', true %}
{% for staffer in leads %}
{{ staffer }}
{% endfor %}

## Predavanja

{% assign instructors = site.staffers | where: 'lectures', true %}
{% for staffer in instructors %}
{{ staffer }}
{% endfor %}

{% assign teaching_assistants = site.staffers | where: 'labs', true %}
{% assign num_teaching_assistants = teaching_assistants | size %}
{% if num_teaching_assistants != 0 %}
## Laboratorijske vježbe

{% for staffer in teaching_assistants %}
{{ staffer }}
{% endfor %}
{% endif %}

{% assign in_memoriam = site.staffers | where: 'memoriam', true %}
{% assign num_in_memoriam = in_memoriam | size %}
{% if num_in_memoriam != 0 %}
## In memoriam

{% for staffer in in_memoriam %}
{{ staffer }}
{% endfor %}
{% endif %}
