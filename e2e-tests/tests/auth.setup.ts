import {
  expect,
  test as setup,
} from '@playwright/test';

const authFile = 'playwright/.auth/user.json';

setup('authenticate', async ({ page }) => {
  await page.goto('/admin/');
  await page.getByLabel('Email address').fill('test@example.com');
  await page.getByRole('button', { name: 'Sign in' }).click();

  const passwordField = await page.getByLabel('Password');
  await expect(passwordField).toBeVisible();
  await passwordField.fill('test');
  await page.getByRole('button', { name: 'Sign in' }).click();

  await page.waitForURL('/admin/');
  await page.context().storageState({ path: authFile });
});
