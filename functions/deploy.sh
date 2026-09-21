#!/bin/bash
echo "Copying leo_simulator to functions dir for deployment..."
cp -r ../leo_simulator ./leo_simulator
firebase deploy --only functions
rm -rf ./leo_simulator
