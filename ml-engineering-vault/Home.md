---
type: moc
title: Home
created: 2026-04-22
---

# ML Engineering Vault

## Recently Updated

```dataview
TABLE type AS Type, libs AS Libraries, updated AS Updated
FROM -"_templates" AND -"_assets"
WHERE type != null
SORT updated DESC
LIMIT 15
```

## By Type

### Snippets

```dataview
TABLE libs AS Libraries, tags AS Tags
FROM "snippets"
WHERE type = "snippet"
SORT updated DESC
```

### Cheat Sheets

```dataview
TABLE libs AS Libraries, tags AS Tags
FROM "cheatsheets"
WHERE type = "cheatsheet"
SORT title ASC
```

### Concepts

```dataview
TABLE libs AS Libraries, tags AS Tags
FROM "concepts"
WHERE type = "concept"
SORT title ASC
```

### Recipes

```dataview
TABLE libs AS Libraries, tags AS Tags
FROM "recipes"
WHERE type = "recipe"
SORT updated DESC
```

### Troubleshooting

```dataview
TABLE libs AS Libraries, tags AS Tags
FROM "troubleshooting"
WHERE type = "troubleshooting"
SORT updated DESC
```

## By Library

### pandas

```dataview
LIST
FROM -"_templates" AND -"_assets"
WHERE contains(libs, "pandas")
SORT type ASC
```

### pytorch

```dataview
LIST
FROM -"_templates" AND -"_assets"
WHERE contains(libs, "pytorch")
SORT type ASC
```

### scikit-learn

```dataview
LIST
FROM -"_templates" AND -"_assets"
WHERE contains(libs, "scikit-learn")
SORT type ASC
```

### matplotlib / seaborn

```dataview
LIST
FROM -"_templates" AND -"_assets"
WHERE contains(libs, "matplotlib") OR contains(libs, "seaborn")
SORT type ASC
```

### statsmodels

```dataview
LIST
FROM -"_templates" AND -"_assets"
WHERE contains(libs, "statsmodels")
SORT type ASC
```

### xgboost / catboost

```dataview
LIST
FROM -"_templates" AND -"_assets"
WHERE contains(libs, "xgboost") OR contains(libs, "catboost")
SORT type ASC
```

### spacy / nltk

```dataview
LIST
FROM -"_templates" AND -"_assets"
WHERE contains(libs, "spacy") OR contains(libs, "nltk")
SORT type ASC
```

## Cross-Reference: Related Notes

```dataview
TABLE related AS "Links To", libs AS Libraries
FROM -"_templates" AND -"_assets"
WHERE type != null AND related != null AND length(related) > 0
SORT title ASC
```

## Orphan Notes (no related links yet)

```dataview
LIST
FROM -"_templates" AND -"_assets"
WHERE type != null AND type != "moc" AND (!related OR length(related) = 0 OR all(related, (r) => r = "[[]]"))
SORT updated DESC
```
