import type { BrowserContext, Page } from 'playwright-core';
import { test } from '@playwright/test';
import { baseURL } from '../playwright.config';

const USER_EMAIL = process.env.DATAROBOT_USER || 'buzok-ci-agents@datarobot.com';
const USER_PASSWORD = process.env.DATAROBOT_PASSWORD!;
const TIMEOUT = 5 * 60 * 1000;

test.describe('auth flow', () => {
  test.use({ testIdAttribute: 'test-id' });

  let context: BrowserContext;
  let page: Page;

  test.beforeAll(async ({ browser }) => {
    context = await browser.newContext({
      serviceWorkers: 'block',
    });
    page = await context.newPage();
  });

  test('login if env is not localhost', async () => {
    const origin = new URL(baseURL).origin;
    if (origin.includes('localhost')) {
      test.skip();
    }
    test.setTimeout(TIMEOUT);

    await page.goto(origin, { waitUntil: 'networkidle' });
    await page.waitForURL('**/login**');
    await page.getByTestId('email-field').type(USER_EMAIL);
    await page.getByTestId('password-field').type(USER_PASSWORD);
    await page.getByTestId('sign-in-button').click();

    await page.waitForURL(origin); // wait until login will be fully done
    await page.goto(baseURL, { waitUntil: 'networkidle' });
    await page.context().storageState({
      path: './e2e/storageState.json',
    });
  });
});
