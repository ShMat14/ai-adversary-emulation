// Neo4j Desktop: copy the three CSV files into the database's import folder first.
CREATE CONSTRAINT technique_name IF NOT EXISTS FOR (t:Technique) REQUIRE t.name IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///techniques.csv' AS r
MERGE (t:Technique {name: r.name})
SET t.id = toInteger(r.id), t.caption = r.caption, t.attack_id = r.attack_id, t.tactic = r.tactic,
    t.campaigns = toInteger(r.campaigns), t.share = toFloat(r.share),
    t.uses = toInteger(r.uses), t.repeats = toInteger(r.repeats), t.band = r.band;

LOAD CSV WITH HEADERS FROM 'file:///next.csv' AS r
MATCH (a:Technique {name: r.from}), (b:Technique {name: r.to})
MERGE (a)-[e:NEXT]->(b) SET e.count = toInteger(r.count);

LOAD CSV WITH HEADERS FROM 'file:///together.csv' AS r
MATCH (a:Technique {name: r.a}), (b:Technique {name: r.b})
MERGE (a)-[e:WITH]->(b) SET e.campaigns = toInteger(r.campaigns), e.jaccard = toFloat(r.jaccard);
