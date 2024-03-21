// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { ChatFollowup, Command } from "vscode";

export interface CommandRunningStatus {
  result: "success" | "fail" | "no run";
  time: Date;
}

export interface MachineStatus {
  firstInstalled: boolean; // if TTK is first installed
  resultOfPrerequistes?: string; // the result of the prerequisites check
  m365LoggedIn: boolean; // if the user has logged in M365
  azureLoggedIn: boolean; // if the user has logged in Azure
}

export interface ProjectActionStatus {
  debug: CommandRunningStatus; // the status of last debugging
  provision: CommandRunningStatus; // the status of last provisioning
  deploy: CommandRunningStatus; // the status of last deploying
  publish: CommandRunningStatus; // the status of last publishing

  openReadMe: CommandRunningStatus; // the status of last showing/summarizing readme
}

export interface WholeStatus {
  machineStatus: MachineStatus;
  projectOpened?: {
    path: string; // the path of the opened app
    projectId?: string; // the project id of the opened app, it is from teamsapp.yml
    codeModifiedTime: {
      source: Date; // the time when the source code is modified
      infra: Date; // the time when the infra is modified
    };
    actionStatus: ProjectActionStatus;
    readmeContent?: string; // the content of the readme file
    launchJSONContent?: string; // the content of the .vscode/launch.json
  };
}

export type Condition = (status: WholeStatus) => boolean;
export type DescripitionFunc = (status: WholeStatus) => string;

export interface NextStep {
  title: string;
  description: string | DescripitionFunc;
  docLink?: string;
  commands: Command[];
  followUps: ChatFollowup[];
  condition: Condition;
  priority: 0 | 1 | 2; // 0: high, 1: medium, 2: low
}
