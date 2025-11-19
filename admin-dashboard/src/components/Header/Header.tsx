import React, { FC } from 'react'
import { HeaderProps } from '../../types/detection'
import './Header.css'

const Header: FC<HeaderProps> = ({ onSettingsClick, onHelpClick }) => {
  return React.createElement(
    'header',
    { className: 'header' },
    React.createElement(
      'div',
      { className: 'header-left' },
      React.createElement('h1', { className: 'header-title' }, 'Dashboard')
    ),
    React.createElement(
      'div',
      { className: 'header-right' },
      React.createElement(
        'button',
        {
          className: 'header-icon-button',
          onClick: onSettingsClick,
          title: 'Settings',
        },
        React.createElement('span', null, '⚙')
      ),
      React.createElement(
        'button',
        {
          className: 'header-icon-button',
          onClick: onHelpClick,
          title: 'Help',
        },
        React.createElement('span', null, '?')
      )
    )
  )
}

export default Header
