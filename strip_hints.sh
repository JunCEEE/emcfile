#!/bin/bash

for f in $(find emcfile/ -name  "*.py");
do
    echo $f
	sed -i 's/\/,//g' $f
    strip-hints $f --inplace --to-empty 
    sed -i -e '/^from typing.*/a from typing import List' \
        -e 's/\/,//g' \
        -e 's/import numpy.typing.*//g' \
        -e 's/npt.NDArray/List/g'\
		-e 's/^from __future__ import annotations//'\
		$f
done

