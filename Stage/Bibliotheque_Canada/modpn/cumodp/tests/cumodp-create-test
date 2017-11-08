#! /bin/sh
# Check if the name of the test is created
if [ -z "$1" ];then
	echo "Usage: cumodp-create-test.sh [NAME OF YOUR TEST]"
	exit;
fi
testname=$1

cp .test_script.template ${testname}.sh || exit $?
sed -i "s/TARGET=\"\"/TARGET=\"${testname}\"/g" ${testname}.sh
chmod +x ${testname}.sh || exit $?

echo "Test: ["$testname"] Created"
echo "Due to time constraint I didn't make everything automated."
echo "Please change the fields in ${testname}.sh to run your test"

echo "["$(date)"] TEST: [$testname] CREATED BY <"$(whoami)">" >> CHANGELOG
