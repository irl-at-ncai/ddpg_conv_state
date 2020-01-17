#!/bin/sh

AGENT_NAME="example_agent"
SCRIPT_PATH=$(dirname `which $0`)

usage()
{
    echo "Usage:"
    echo "./init_agent.sh -n=<AGENT_NAME>"
    echo ""
    echo " -h | --help : Displays the help"
    echo " -n | --agent-name : Agent name in lower case letters (" $AGENT_NAME ")"
    echo ""
}

while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | awk -F= '{print $2}'`
    case $PARAM in
        -h | --help)
            usage
            exit
            ;;
        -n  | --agent-name )
            AGENT_NAME=$VALUE
            ;;
        *)
            echo "ERROR: unknown parameter \"$PARAM\""
            usage
            exit 1
            ;;
    esac
    shift
done

if [ "$AGENT_NAME" = "example_agent" ]; then
    echo "Please specify agent-name argument."
    usage
    exit 1
fi

echo "${SCRIPT_PATH}/../src/rl_agents/$AGENT_NAME"
if [ -d "${SCRIPT_PATH}/../src/rl_agents/$AGENT_NAME" ]; then
  echo "An agent with the name ${AGENT_NAME} already exists."
  exit
else
	mkdir -p "${SCRIPT_PATH}/../src/rl_agents/$AGENT_NAME/cfg"
	cp "${SCRIPT_PATH}/../cfg/agent_base.yaml" "${SCRIPT_PATH}/../src/rl_agents/$AGENT_NAME/cfg/${AGENT_NAME}_1.yaml"
    sed -i "s/agent_base/${AGENT_NAME}/g" "${SCRIPT_PATH}/../src/rl_agents/$AGENT_NAME/cfg/${AGENT_NAME}_1.yaml"
    cp "${SCRIPT_PATH}/../src/rl_agents/tests/test_ddpg.py" "${SCRIPT_PATH}/../src/rl_agents/tests/test_${AGENT_NAME}.py"
    sed -i "s/ddpg/${AGENT_NAME}/g" "${SCRIPT_PATH}/../src/rl_agents/tests/test_${AGENT_NAME}.py"
    cp "${SCRIPT_PATH}/../launch/agent_base.launch" "${SCRIPT_PATH}/../launch/${AGENT_NAME}.launch"
    sed -i "s/agent_base/${AGENT_NAME}/g" "${SCRIPT_PATH}/../launch/${AGENT_NAME}.launch"
    cp "${SCRIPT_PATH}/../launch/test_ddpg.launch" "${SCRIPT_PATH}/../launch/test_${AGENT_NAME}.launch"
    sed -i "s/ddpg/${AGENT_NAME}/g" "${SCRIPT_PATH}/../launch/test_${AGENT_NAME}.launch"
fi