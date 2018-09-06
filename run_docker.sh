#!/bin/sh

docker build -t trend-hearts .
docker tag trend-hearts ai.registry.trendmicro.com/274/trend-hearts:rank
docker tag trend-hearts ai.registry.trendmicro.com/274/trend-hearts:practice
docker push ai.registry.trendmicro.com/274/trend-hearts:rank
docker push ai.registry.trendmicro.com/274/trend-hearts:rank
