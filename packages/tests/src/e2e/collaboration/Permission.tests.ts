// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 * @author Bowen Song <bowen.song@microsoft.com>
 */

import { expect } from "chai";
import * as fs from "fs-extra";
import path from "path";
import {
  cleanUp,
  execAsync,
  execAsyncWithRetry,
  getTestFolder,
  getUniqueAppName,
  removeTeamsAppExtendToM365,
} from "../commonUtils";

import { it } from "@microsoft/extra-shot-mocha";
import { CliHelper } from "../../commonlib/cliHelper";

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

describe("Collaboration", function () {
  const testFolder = getTestFolder();
  let appName = getUniqueAppName();
  let projectPath = path.resolve(testFolder, appName);
  const collaborator = process.env["M365_ACCOUNT_COLLABORATOR"];
  const creator = process.env["M365_ACCOUNT_NAME"];

  it(
    "Collaboration: CLI with permission status and permission grant",
    { testPlanCaseId: 10753319, author: "bowen.song@microsoft.com" },
    async function () {
      while (await fs.pathExists(projectPath)) {
        appName = getUniqueAppName();
        projectPath = path.resolve(testFolder, appName);
      }

      // new a project
      await execAsync(
        `teamsfx new --interactive false --capabilities sso-launch-page --app-name ${appName}`,
        {
          cwd: testFolder,
          env: process.env,
          timeout: 0,
        }
      );
      console.log(`[Successfully] scaffold to ${projectPath}`);
      const filePath = path.join(projectPath, "teamsapp.yml");
      removeTeamsAppExtendToM365(filePath);

      // provision
      await CliHelper.provisionProject(projectPath);
      await sleep(10000);

      // Grant Permission
      const grantCollaboratorResult = await execAsyncWithRetry(
        `teamsfx permission grant --email ${collaborator} --env dev --teams-app-manifest ${projectPath}/appPackage/manifest.json --aad-app-manifest ${projectPath}/aad.manifest.json --interactive false`,
        {
          cwd: projectPath,
          env: process.env,
          timeout: 0,
        }
      );

      expect(grantCollaboratorResult.stdout).to.contains(
        "Owner permission has been granted to Azure AD App"
      );
      expect(grantCollaboratorResult.stdout).to.contains(
        "Administrator permission has been granted to Teams App"
      );
      console.log("[Successfully] grant permission");

      // Check Permission
      const checkPermissionResult = await execAsyncWithRetry(
        `teamsfx permission status --env dev --interactive false --teams-app-manifest ./appPackage/manifest.json --aad-app-manifest ./aad.manifest.json`,
        {
          cwd: projectPath,
          env: process.env,
          timeout: 0,
        }
      );
      console.log(`check permission: ${checkPermissionResult.stdout}`);

      expect(checkPermissionResult.stdout).to.contains(
        "Resource Name: Azure AD App, Permission: Owner"
      );
      expect(checkPermissionResult.stdout).to.contains(
        "Resource Name: Teams App, Permission: Administrator"
      );
      console.log("[Successfully] check permission");

      const listCollaboratorResult = await execAsync(
        `teamsfx permission status --list-all-collaborators --env dev --teams-app-manifest ${projectPath}/appPackage/manifest.json --aad-app-manifest ${projectPath}/aad.manifest.json --interactive false`,
        {
          cwd: projectPath,
          env: process.env,
          timeout: 0,
        }
      );

      // Check collaborator.
      // When collaborator account is guest account in the tenant. Account name pattern will change.
      // e.g. Guest account "account@example.com" will appear as "account_example#EXT#@exampleTenant.onmicrosoft.com" under tenant "exampleTenant".
      // Thus here will check the account name only.
      expect(listCollaboratorResult.stdout).to.contains(
        `Account used to check: ${creator?.split("@")[0]}`
      );
      expect(listCollaboratorResult.stdout).to.contains(
        `Teams App Owner: ${collaborator?.split("@")[0]}`
      );

      console.log("[Successfully] list collaborator");
    }
  );

  after(async () => {
    // clean up
    await cleanUp(appName, projectPath, true, false, false);
  });
});
